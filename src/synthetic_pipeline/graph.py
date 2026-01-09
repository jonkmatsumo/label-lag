"""Graph network generator for User, Device, and Account relationships."""

import uuid
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
from faker import Faker
from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """Types of nodes in the graph."""

    USER = "user"
    DEVICE = "device"
    ACCOUNT = "account"
    IP_ADDRESS = "ip_address"


class EdgeType(str, Enum):
    """Types of edges (relationships) in the graph."""

    USES_DEVICE = "uses_device"
    HAS_ACCOUNT = "has_account"
    USES_IP = "uses_ip"
    TRANSFERS_TO = "transfers_to"


class Node(BaseModel):
    """Represents a node in the graph."""

    id: str = Field(..., description="Unique node identifier")
    type: NodeType = Field(..., description="Type of the node")
    properties: dict = Field(
        default_factory=dict, description="Additional node properties"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="Node creation timestamp"
    )

    def to_csv_row(self) -> dict:
        """Convert node to a flat dictionary for CSV export."""
        return {
            "id": self.id,
            "type": self.type.value,
            "created_at": self.created_at.isoformat(),
            **{f"prop_{k}": v for k, v in self.properties.items()},
        }


class Edge(BaseModel):
    """Represents an edge (relationship) in the graph."""

    id: str = Field(..., description="Unique edge identifier")
    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    type: EdgeType = Field(..., description="Type of the relationship")
    properties: dict = Field(
        default_factory=dict, description="Additional edge properties"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="Edge creation timestamp"
    )

    def to_csv_row(self) -> dict:
        """Convert edge to a flat dictionary for CSV export."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type.value,
            "created_at": self.created_at.isoformat(),
            **{f"prop_{k}": v for k, v in self.properties.items()},
        }


class GraphData(BaseModel):
    """Container for graph nodes and edges."""

    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)

    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        self.nodes.append(node)

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)

    def merge(self, other: "GraphData") -> "GraphData":
        """Merge another GraphData into this one."""
        return GraphData(
            nodes=self.nodes + other.nodes,
            edges=self.edges + other.edges,
        )

    def get_nodes_by_type(self, node_type: NodeType) -> list[Node]:
        """Get all nodes of a specific type."""
        return [n for n in self.nodes if n.type == node_type]

    def get_edges_by_type(self, edge_type: EdgeType) -> list[Edge]:
        """Get all edges of a specific type."""
        return [e for e in self.edges if e.type == edge_type]

    def to_node_csv_rows(self) -> list[dict]:
        """Convert all nodes to CSV-compatible rows."""
        return [n.to_csv_row() for n in self.nodes]

    def to_edge_csv_rows(self) -> list[dict]:
        """Convert all edges to CSV-compatible rows."""
        return [e.to_csv_row() for e in self.edges]


class GraphNetworkGenerator:
    """Generator for graph network data with fraud patterns.

    Generates relationships between Users, Devices, Accounts, and IP addresses
    with support for both legitimate patterns and fraud scenarios.

    Fraud patterns supported:
    - Device sharing fraud rings [cite: 106]
    - IP recycling clusters [cite: 93]
    - Fund kiting cycles [cite: 107]
    """

    def __init__(self, seed: int | None = None):
        """Initialize the generator.

        Args:
            seed: Random seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)
        self.faker = Faker()
        if seed is not None:
            Faker.seed(seed)

    def _generate_id(self, prefix: str) -> str:
        """Generate a unique ID with prefix."""
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    def _generate_device_fingerprint(self) -> str:
        """Generate a realistic device fingerprint."""
        return uuid.uuid4().hex

    def _generate_ip_address(self) -> str:
        """Generate a random IP address."""
        return self.faker.ipv4()

    def _generate_user_node(self, base_time: datetime | None = None) -> Node:
        """Generate a user node with PII."""
        return Node(
            id=self._generate_id("user"),
            type=NodeType.USER,
            properties={
                "name": self.faker.name(),
                "email": self.faker.email(),
                "phone": self.faker.phone_number(),
            },
            created_at=base_time or datetime.now(),
        )

    def _generate_device_node(self, base_time: datetime | None = None) -> Node:
        """Generate a device node."""
        device_types = ["mobile", "desktop", "tablet"]
        os_types = ["iOS", "Android", "Windows", "macOS", "Linux"]
        return Node(
            id=self._generate_id("device"),
            type=NodeType.DEVICE,
            properties={
                "fingerprint": self._generate_device_fingerprint(),
                "device_type": self.rng.choice(device_types),
                "os": self.rng.choice(os_types),
                "user_agent": self.faker.user_agent(),
            },
            created_at=base_time or datetime.now(),
        )

    def _generate_account_node(self, base_time: datetime | None = None) -> Node:
        """Generate an account node."""
        return Node(
            id=self._generate_id("account"),
            type=NodeType.ACCOUNT,
            properties={
                "account_number": self.faker.bban(),
                "account_type": self.rng.choice(["checking", "savings"]),
                "balance": float(round(self.rng.uniform(100, 50000), 2)),
            },
            created_at=base_time or datetime.now(),
        )

    def _generate_ip_node(
        self, ip_address: str | None = None, base_time: datetime | None = None
    ) -> Node:
        """Generate an IP address node."""
        return Node(
            id=self._generate_id("ip"),
            type=NodeType.IP_ADDRESS,
            properties={
                "address": ip_address or self._generate_ip_address(),
                "is_vpn": bool(self.rng.choice([True, False], p=[0.1, 0.9])),
                "country": self.faker.country_code(),
            },
            created_at=base_time or datetime.now(),
        )

    def generate_legitimate_user_network(self, count: int = 1) -> GraphData:
        """Generate legitimate user networks.

        Each user has 1-2 devices, 1-2 accounts, and uses 1-3 unique IPs.

        Args:
            count: Number of users to generate.

        Returns:
            GraphData containing nodes and edges.
        """
        graph = GraphData()

        for _ in range(count):
            base_time = datetime.now() - timedelta(days=int(self.rng.integers(1, 365)))
            user = self._generate_user_node(base_time)
            graph.add_node(user)

            # 1-2 devices per user (legitimate)
            num_devices = int(self.rng.integers(1, 3))
            for _ in range(num_devices):
                device = self._generate_device_node(base_time)
                graph.add_node(device)
                graph.add_edge(
                    Edge(
                        id=self._generate_id("edge"),
                        source_id=user.id,
                        target_id=device.id,
                        type=EdgeType.USES_DEVICE,
                        created_at=base_time,
                    )
                )

            # 1-2 accounts per user
            num_accounts = int(self.rng.integers(1, 3))
            for _ in range(num_accounts):
                account = self._generate_account_node(base_time)
                graph.add_node(account)
                graph.add_edge(
                    Edge(
                        id=self._generate_id("edge"),
                        source_id=user.id,
                        target_id=account.id,
                        type=EdgeType.HAS_ACCOUNT,
                        created_at=base_time,
                    )
                )

            # 1-3 unique IPs per user
            num_ips = int(self.rng.integers(1, 4))
            for _ in range(num_ips):
                ip_node = self._generate_ip_node(base_time=base_time)
                graph.add_node(ip_node)
                graph.add_edge(
                    Edge(
                        id=self._generate_id("edge"),
                        source_id=user.id,
                        target_id=ip_node.id,
                        type=EdgeType.USES_IP,
                        created_at=base_time,
                    )
                )

        return graph

    def generate_device_sharing_fraud(
        self,
        num_identities: int | None = None,
        days_window: int = 7,
    ) -> GraphData:
        """Generate a device sharing fraud ring.

        Creates a pattern where one device is used by >5 unique identities
        within a specified time window (degree centrality anomaly) [cite: 106].

        Args:
            num_identities: Number of identities sharing the device.
                           Defaults to random 6-15.
            days_window: Time window in days. Defaults to 7.

        Returns:
            GraphData containing the fraud ring.
        """
        if num_identities is None:
            num_identities = int(self.rng.integers(6, 16))

        graph = GraphData()
        base_time = datetime.now()

        # Create the shared device (high degree centrality)
        shared_device = self._generate_device_node(base_time)
        shared_device.properties["is_fraud_indicator"] = True
        shared_device.properties["fraud_type"] = "device_sharing"
        graph.add_node(shared_device)

        # Create multiple users sharing this device
        for i in range(num_identities):
            # Spread user creation across the time window
            user_time = base_time - timedelta(
                days=float(self.rng.uniform(0, days_window))
            )
            user = self._generate_user_node(user_time)
            user.properties["is_fraud_indicator"] = True
            user.properties["fraud_ring_id"] = shared_device.id
            graph.add_node(user)

            # Connect user to shared device
            graph.add_edge(
                Edge(
                    id=self._generate_id("edge"),
                    source_id=user.id,
                    target_id=shared_device.id,
                    type=EdgeType.USES_DEVICE,
                    properties={"access_count": int(self.rng.integers(1, 20))},
                    created_at=user_time,
                )
            )

            # Each fraudulent user also has an account
            account = self._generate_account_node(user_time)
            account.properties["is_fraud_indicator"] = True
            graph.add_node(account)
            graph.add_edge(
                Edge(
                    id=self._generate_id("edge"),
                    source_id=user.id,
                    target_id=account.id,
                    type=EdgeType.HAS_ACCOUNT,
                    created_at=user_time,
                )
            )

        return graph

    def generate_ip_recycling_fraud(
        self,
        num_users: int | None = None,
        is_vpn: bool = True,
    ) -> GraphData:
        """Generate an IP recycling fraud cluster.

        Creates a pattern where multiple users share the same IP address
        or VPN endpoint [cite: 93].

        Args:
            num_users: Number of users sharing the IP. Defaults to random 5-20.
            is_vpn: Whether the shared IP is a VPN endpoint.

        Returns:
            GraphData containing the fraud cluster.
        """
        if num_users is None:
            num_users = int(self.rng.integers(5, 21))

        graph = GraphData()
        base_time = datetime.now()

        # Create the shared IP address
        shared_ip = self._generate_ip_node(base_time=base_time)
        shared_ip.properties["is_vpn"] = is_vpn
        shared_ip.properties["is_fraud_indicator"] = True
        shared_ip.properties["fraud_type"] = "ip_recycling"
        shared_ip.properties["user_count"] = num_users
        graph.add_node(shared_ip)

        # Create users sharing this IP
        for _ in range(num_users):
            user_time = base_time - timedelta(hours=float(self.rng.uniform(0, 168)))
            user = self._generate_user_node(user_time)
            user.properties["is_fraud_indicator"] = True
            user.properties["fraud_cluster_ip"] = shared_ip.properties["address"]
            graph.add_node(user)

            # Connect user to shared IP
            graph.add_edge(
                Edge(
                    id=self._generate_id("edge"),
                    source_id=user.id,
                    target_id=shared_ip.id,
                    type=EdgeType.USES_IP,
                    properties={"session_count": int(self.rng.integers(1, 50))},
                    created_at=user_time,
                )
            )

            # Each user has their own device (to avoid overlap with device sharing)
            device = self._generate_device_node(user_time)
            graph.add_node(device)
            graph.add_edge(
                Edge(
                    id=self._generate_id("edge"),
                    source_id=user.id,
                    target_id=device.id,
                    type=EdgeType.USES_DEVICE,
                    created_at=user_time,
                )
            )

            # Each user has an account
            account = self._generate_account_node(user_time)
            account.properties["is_fraud_indicator"] = True
            graph.add_node(account)
            graph.add_edge(
                Edge(
                    id=self._generate_id("edge"),
                    source_id=user.id,
                    target_id=account.id,
                    type=EdgeType.HAS_ACCOUNT,
                    created_at=user_time,
                )
            )

        return graph

    def generate_kiting_cycle(
        self,
        cycle_length: int = 3,
        num_cycles: int = 1,
        transfer_amount: float | None = None,
    ) -> GraphData:
        """Generate a fund kiting cycle pattern.

        Creates a pattern where funds move in a cycle:
        Account A -> B -> C -> A [cite: 107].

        Args:
            cycle_length: Number of accounts in the cycle. Minimum 3.
            num_cycles: Number of complete cycles to generate.
            transfer_amount: Amount transferred in each hop.
                           Defaults to random 1000-10000.

        Returns:
            GraphData containing the kiting cycle.
        """
        cycle_length = max(3, cycle_length)
        if transfer_amount is None:
            transfer_amount = float(round(self.rng.uniform(1000, 10000), 2))

        graph = GraphData()
        base_time = datetime.now()

        # Create accounts and users for the cycle
        accounts: list[Node] = []
        users: list[Node] = []

        for i in range(cycle_length):
            user_time = base_time - timedelta(days=float(self.rng.uniform(30, 365)))

            user = self._generate_user_node(user_time)
            user.properties["is_fraud_indicator"] = True
            user.properties["fraud_type"] = "kiting"
            user.properties["cycle_position"] = i
            graph.add_node(user)
            users.append(user)

            account = self._generate_account_node(user_time)
            account.properties["is_fraud_indicator"] = True
            account.properties["fraud_type"] = "kiting"
            account.properties["cycle_position"] = i
            graph.add_node(account)
            accounts.append(account)

            # Link user to account
            graph.add_edge(
                Edge(
                    id=self._generate_id("edge"),
                    source_id=user.id,
                    target_id=account.id,
                    type=EdgeType.HAS_ACCOUNT,
                    created_at=user_time,
                )
            )

        # Create the cyclic transfer pattern
        for cycle_num in range(num_cycles):
            cycle_start_time = base_time - timedelta(days=float(self.rng.uniform(0, 7)))

            for i in range(cycle_length):
                source_account = accounts[i]
                target_account = accounts[(i + 1) % cycle_length]

                # Time between transfers (hours)
                transfer_time = cycle_start_time + timedelta(
                    hours=float(i * self.rng.uniform(1, 24))
                )

                graph.add_edge(
                    Edge(
                        id=self._generate_id("edge"),
                        source_id=source_account.id,
                        target_id=target_account.id,
                        type=EdgeType.TRANSFERS_TO,
                        properties={
                            "amount": transfer_amount,
                            "cycle_number": cycle_num + 1,
                            "hop_number": i + 1,
                            "is_kiting": True,
                        },
                        created_at=transfer_time,
                    )
                )

        return graph

    def generate_mixed_network(
        self,
        num_legitimate_users: int = 100,
        num_device_sharing_rings: int = 2,
        num_ip_clusters: int = 2,
        num_kiting_cycles: int = 1,
    ) -> GraphData:
        """Generate a mixed network with both legitimate and fraudulent patterns.

        Args:
            num_legitimate_users: Number of legitimate user networks.
            num_device_sharing_rings: Number of device sharing fraud rings.
            num_ip_clusters: Number of IP recycling fraud clusters.
            num_kiting_cycles: Number of kiting cycle patterns.

        Returns:
            GraphData containing all patterns.
        """
        graph = GraphData()

        # Generate legitimate users
        graph = graph.merge(self.generate_legitimate_user_network(num_legitimate_users))

        # Generate device sharing fraud rings
        for _ in range(num_device_sharing_rings):
            graph = graph.merge(self.generate_device_sharing_fraud())

        # Generate IP recycling clusters
        for _ in range(num_ip_clusters):
            graph = graph.merge(self.generate_ip_recycling_fraud())

        # Generate kiting cycles
        for _ in range(num_kiting_cycles):
            graph = graph.merge(self.generate_kiting_cycle())

        return graph
